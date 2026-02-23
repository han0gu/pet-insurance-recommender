from langchain_core.documents import Document

chunk = Document(
    page_content=('인 경우에 회사는 14일(보험기간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최고\n'
 '(독촉)기간(납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은\n'
 '그 다음 날까지로 합니다)으로 정하여 아래 사항에 대하여 서면(등기우편 등), 전화(음\n'
 '성녹음) 또는 전자문서 등으로 알려드립니다. 다만, 해지 전에 발생한 보험금 지급사유\n'
 '에 대하여 회사는 보상하여 드립니다.- 1. 계약자(보험수익자와 계약자가 다른 경우 보험수익자를 포함합니다)에게 납입최고\n'
 '- (독촉)기간 내에 연체보험료를 납입하여야 한다는 내용'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000089',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
