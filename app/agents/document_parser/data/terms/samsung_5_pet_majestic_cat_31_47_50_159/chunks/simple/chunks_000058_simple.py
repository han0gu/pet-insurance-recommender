from langchain_core.documents import Document

chunk = Document(
    page_content=('제3관 계약자의 계약 전 알릴 의무 등\n'
 '제 16조 (계약 전 알릴 의무)\n'
 '계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건강진단할 때를 말합니다) 청약 서에서 질문한 사항에 대하여 알고 있는 사실을 '
 '반드시 사실대로 알려야(이하「계약 전 알릴 의무」라 하며, 상법상「고지의무」와 같습니다) 합니다. 다만, 진단계약의 경우 의 료법 '
 '제3조(의료기관)의 규정에 따른 종합병원과 병원에서 직장 또는 개인이 실시한 건강 진단서 사본 등 건강상태를 판단할 수 있는 자료로 '
 '건강진단을 대신할 수 있습니다.\n'
 '<관련법규>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000058',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
