from langchain_core.documents import Document

chunk = Document(
    page_content=("추가 보험<br>료를 받기 전에 생긴 손해는 보상하여 드리지 않습니다.</p><h1 id='7' "
 "style='font-size:14px'>제4조(준용규정)</h1><br><p id='8' data-category='paragraph' "
 "style='font-size:14px'>이 추가특별약관에 정하지 않은 사항은 보통약관 및 단체계약 특별약관을 "
 "따릅니다.</p><footer id='9' style='font-size:14px'>- 41 -</footer><h1 id='10' "
 "style='font-size:18px'>지정대리청구서비스"),
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
 'indexing': {'chunk_id': 'chunk_000342',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
