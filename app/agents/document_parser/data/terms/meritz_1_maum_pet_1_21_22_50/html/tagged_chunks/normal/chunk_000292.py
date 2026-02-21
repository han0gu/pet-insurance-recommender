from langchain_core.documents import Document

chunk = Document(
    page_content=("지급하지 않는 사유)</h1><br><p id='14' data-category='paragraph' "
 "style='font-size:14px'>회사는 보통약관 제4조(보험금의 지급사유) 에도 불구하고 이 특별약관에 따라 "
 "반려동물<br>치료비 보험금을 보상하여 드리지 않습니다.</p><h1 id='15' "
 "style='font-size:14px'>제2조(준용규정)</h1><br><p id='16' "
 "data-category='paragraph' style='font-size:14px'>이 특별약관에 정하지 않은 사항은 보통약관을"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000292',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
