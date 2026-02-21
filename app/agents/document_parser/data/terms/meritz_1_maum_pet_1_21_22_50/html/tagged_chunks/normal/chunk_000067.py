from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제3자는 수의사법 제2조(정의)에 규정한 동물병원 소속의 수의사 중에서 정하며,</p><footer id='72' "
 "style='font-size:14px'>- 7 -</footer><header id='73' "
 "style='font-size:14px'>보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담합니다.</header><h1 "
 "id='74' style='font-size:14px'>제10조(지급보험금의 계산)</h1><br><p id='75' "
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
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000067',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
