from langchain_core.documents import Document

chunk = Document(
    page_content=('제 3 관 계약자의 계약 전 알릴 의무 등\n'
 '제15조(계약 전 알릴 의무)\n'
 '계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건강진단할 때를 말합니다) 청약 서에서 질문한 사항에 대하여 알고 있는 사실을 '
 '반드시 사실대로 알려야(이하 「계약 전 알릴 의무」라 하며, 상법상「고지의무」와 같습니다) 합니다.\n'
 '【계약 전 알릴 의무】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000054',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
