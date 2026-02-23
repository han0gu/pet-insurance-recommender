from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것</p><br><p id='59' "
 "data-category='paragraph' style='font-size:14px'>④ 제1항에 따라 계약이 해지되고 이로 인하여 "
 '회사가 환급하여야 할 보험료가 있을 때에<br>는 제33조(보험료의 환급)에 따른 보험료를 계약자에게 지급합니다.</p><br><h1 '
 "id='60' style='font-size:14px'>【납입최고(독촉)】</h1><br><p id='61' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
