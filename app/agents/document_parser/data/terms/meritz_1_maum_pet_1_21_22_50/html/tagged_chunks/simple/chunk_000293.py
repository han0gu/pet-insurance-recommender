from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.</p><footer id='17' "
 "style='font-size:14px'>- 33 -</footer><h1 id='18' "
 "style='font-size:18px'>보험료분납 특별약관∥</h1><h1 id='19' "
 "style='font-size:14px'>제1조(보험료의 분납)</h1><br><p id='20' "
 "data-category='paragraph' style='font-size:14px'>계약자는 이 특별약관에 따라 보험료를"),
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
 'indexing': {'chunk_id': 'chunk_000293',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
