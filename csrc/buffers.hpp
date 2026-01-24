#pragma once

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <stdexcept>

#include "utils.hpp"

struct CameraBuffer {
    void* start;
    v4l2_buffer v4l_buf;

    size_t length() const {
        return v4l_buf.length;
    }
};

class CameraRingBuffer{
    private:
        CameraBuffer* buffers;
        v4l2_buffer dq_buffer;
        v4l2_buf_type type;
        v4l2_memory memory;
        int fd;
        int n_buffers;
        bool streaming;

        void cleanup(int count) {
            if (!buffers) {
                return;
            }

            for (int i = 0; i < count; i++) {
                if (buffers[i].start && buffers[i].start != MAP_FAILED) {
                    munmap(buffers[i].start, buffers[i].length());
                }
            }
        }
        
    public:
        explicit CameraRingBuffer(
            int _fd, 
            int _n_buffers = 3, 
            v4l2_buf_type _type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
            v4l2_memory _memory = V4L2_MEMORY_MMAP //memory mapping
        ) : buffers(nullptr), n_buffers(_n_buffers), fd(_fd), type(_type), memory(_memory), streaming(false){
            clear(&dq_buffer);
        }

        ~CameraRingBuffer() {
            try { 
                stop_streaming(); 
            } catch(...){}

            cleanup(n_buffers);
            delete[] buffers;
        }

        CameraRingBuffer(const CameraRingBuffer&) = delete;
        CameraRingBuffer& operator=(const CameraRingBuffer&) = delete;

        CameraRingBuffer(CameraRingBuffer&&) = delete;
        CameraRingBuffer& operator=(CameraRingBuffer&&) = delete;

        void init(){
            if (buffers) {
                return;
            }
            v4l2_requestbuffers reqbuff;
            clear(&reqbuff);
            reqbuff.type = type;
            reqbuff.memory = memory;
            reqbuff.count = n_buffers;

            if(ioctl(fd, VIDIOC_REQBUFS, &reqbuff) == -1){
                perror("Camera does NOT support mmap-streaming");
            }
            
            n_buffers = reqbuff.count;
            buffers = new CameraBuffer[n_buffers];

            clear(&dq_buffer);
            dq_buffer.type = type;
            dq_buffer.memory = memory;

            for (int i = 0; i < n_buffers; i++) {
                CameraBuffer& buffer = buffers[i];

                clear(&buffer.v4l_buf);
                buffer.v4l_buf.type = type;
                buffer.v4l_buf.memory = memory;
                buffer.v4l_buf.index = i;

                if (ioctl(fd, VIDIOC_QUERYBUF, &buffer.v4l_buf) == -1) {
                    cleanup(i);
                    throw std::runtime_error("VIDIOC_QUERYBUF failed");
                }

                buffer.start = mmap(NULL, buffer.length(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, buffer.v4l_buf.m.offset);

                if (MAP_FAILED == buffers[i].start) {
                    cleanup(i);
                    throw std::runtime_error("mmap failed");
                }
            }
        }

        void queue_buffer(int i) {
            if(ioctl(fd, VIDIOC_QBUF, &buffers[i].v4l_buf) == -1){
                throw std::runtime_error("Failed to queue buffer in CameraRingBuffer");
            }
        }

        int dequeue_buffer(){
            clear(&dq_buffer);
            dq_buffer.type = type;
            dq_buffer.memory = memory;
            
            if(ioctl(fd, VIDIOC_DQBUF, &dq_buffer) == -1){
                return -1;
            }

            return dq_buffer.index;
        }

        void q_all_buffers(){
            for (int i = 0; i < n_buffers; i++){
                queue_buffer(i);
            }
        }

        bool dq_and_q_buffer(){
            int idx = dequeue_buffer();

            if (idx == -1){
                return false;
            }

            queue_buffer(idx);
            return true;
        }

        void start_streaming(){
            if(streaming){
                return;
            }

            init();
            q_all_buffers();
            
            if(ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
                perror("Failed to begin camera stream");
                 throw std::runtime_error("Failed to begin camera stream");
            }

            streaming = true;
        }

        void stop_streaming(){
            if(!streaming){
                return;
            }

            if(ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
                perror("Failed to terminate camera stream");
                throw std::runtime_error("Failed to end camera stream");
            }

            streaming = false;
        }

        bool is_streaming() const {
            return streaming;
        }

        size_t get_n_buffers() const {
            return n_buffers;
        }

        void* buffer_start(const int i) {
            return buffers[i].start;
        }

        size_t buffer_length(const int i) const {
            return buffers[i].length();
        }
};
