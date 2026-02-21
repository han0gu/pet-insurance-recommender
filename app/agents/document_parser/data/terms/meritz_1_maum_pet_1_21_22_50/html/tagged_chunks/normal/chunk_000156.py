from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 회사가 그 청약을 승낙한 때에는 계약자는 부활(효력회<br>복)을 청약한 날까지의 연체된 보험료에 보험개발원이 공시하는 '
 "월평균 정기예금이율</p><footer id='64' style='font-size:14px'>- 16 -</footer><p "
 "id='65' data-category='paragraph' style='font-size:14px'>+ 1% 범위내에서 각 상품별로 "
 "회사가 정하는 이율로 계산한 금액을 더하여 납입하여<br>야 합니다.</p><br><p id='66' "
 "data-category='list'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000156',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
