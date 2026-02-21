from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타 지정대리청구인이 보험금 등의 수령에 필요하여 제출하는 서류</p><p id='29' "
 "data-category='paragraph' style='font-size:14px'>제7조(준용규정)</p><br><p id='30' "
 "data-category='paragraph' style='font-size:14px'>이 특약에서 정하지 않은 사항에 대하여는 보통약관 "
 "및 해당 특별약관을 따릅니다.</p><footer id='31' style='font-size:14px'>- 43 "
 "-</footer><h1 id='32'"),
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
 'indexing': {'chunk_id': 'chunk_000357',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
