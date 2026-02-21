from langchain_core.documents import Document

chunk = Document(
    page_content=('중<br>인 경우에 회사는 14일(보험기간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최고<br>(독촉)기간(납입최고(독촉)기간의 '
 '마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은<br>그 다음 날까지로 합니다)으로 정하여 아래 사항에 대하여 서면(등기우편 등), '
 '전화(음<br>성녹음) 또는 전자문서 등으로 알려드립니다'),
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
 'indexing': {'chunk_id': 'chunk_000146',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
