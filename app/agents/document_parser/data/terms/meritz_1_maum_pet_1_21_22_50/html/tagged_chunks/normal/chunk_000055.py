from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 같다)을 발급하지 못하며, 「약사법」 제85조제6항에<br>따른 동물용 의약품(이하 “처방대상 동물용 의약품”이라 한다)을 '
 '처방ㆍ투약하<br>지 못한다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 89,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
