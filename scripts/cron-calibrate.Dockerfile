FROM alpine:3.20
RUN apk add --no-cache curl bash
COPY cron-calibrate.sh /usr/local/bin/cron-calibrate.sh
RUN chmod +x /usr/local/bin/cron-calibrate.sh
CMD ["/usr/local/bin/cron-calibrate.sh"]
