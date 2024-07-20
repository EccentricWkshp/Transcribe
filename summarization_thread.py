from PyQt6.QtCore import QThread, pyqtSignal

class SummarizationThread(QThread):
    progress = pyqtSignal(int)
    summary_chunk = pyqtSignal(str)
    summary_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, llm_manager, transcript, summary_type):
        super().__init__()
        self.llm_manager = llm_manager
        self.transcript = transcript
        self.summary_type = summary_type
        self.is_cancelled = False

    def run(self):
        try:
            chunk_size = 4000
            chunks = [self.transcript[i:i+chunk_size] for i in range(0, len(self.transcript), chunk_size)]
            summaries = []
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                if self.is_cancelled:
                    return
                summary = self.llm_manager.summarize(chunk, self.summary_type)
                summaries.append(summary)
                self.progress.emit(int((i + 1) / total_chunks * 50))
                self.summary_chunk.emit(summary)

            combined_summary = "\n\n".join(summaries)
            final_summary = self.llm_manager.summarize(combined_summary, self.summary_type)
            self.progress.emit(100)
            self.summary_complete.emit(final_summary)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        self.is_cancelled = True