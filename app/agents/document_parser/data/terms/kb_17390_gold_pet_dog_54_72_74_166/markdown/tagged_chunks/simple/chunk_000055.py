from langchain_core.documents import Document

chunk = Document(
    page_content=('| 제14조(계약 전 알릴 의무) 계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건강진단할 때를 말합니다) 청약서에서 질문한 '
 '사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 " 계약 전 알릴의무"라 하며, 상법상 "고지의무"와 같습니다) 합니다. '
 '다만, 진단계약 의 경우 의료법 제3조(의료기관)의 규정에 따른 종합병원과 병원에서 직장 또는 개인 이 실시한 건강진단서 사본 등 '
 '건강상태를 판단할 수 있는 자료로 건강진단을 대신할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
