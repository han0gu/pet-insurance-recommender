from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 보험자가 계약당시에 그 사실을 알았거나 중대한 과실로 인하여<br>도<br>알지 못한 때에는 그러하지 '
 '아니하다.<br>성<br>∙ 상법 제651조의2(서면에 의한 질문의 효력)<br>특<br>보험자가 서면으로 질문한 사항은 중요한 사항으로 '
 '추정한다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000822',
              'chunk_char_len': 144,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
