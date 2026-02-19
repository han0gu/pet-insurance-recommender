from langchain_core.documents import Document

chunk = Document(
    page_content=('제3관 계약자의 계약 전 알릴 의무 등\n'
 '제15조(계약 전 알릴 의무) 계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건 강진단할 때를 말합니다) 청약서에서 질문한 사항에 '
 '대하여 알고 있는 사실을 반드시 사실대로 알려야(이하「계약 전 알릴 의무」라 하며, 상법상「고지의무」와 같습니다) 합니 다. 다만, '
 '진단계약의 경우 의료법 제3조(의료기관)의 규정 에 따른 종합병원과 병원에서 직장 또는 개인이 실시한 건 강진단서 사본 등 건강상태를 '
 '판단할 수 있는 자료로 건강 진단을 대신할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 61},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000050',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
