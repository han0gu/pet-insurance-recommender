from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 보험계약자 및 보험료납<br>부자가 법인인 보험계약의 경우에는 보호되지 않습니다.</p><footer id='12' "
 "style='font-size:14px'>- 21 -</footer><h1 id='13' style='font-size:20px'>메리츠 "
 "마음든든 반려동물보험 특별약관</h1><h1 id='14' style='font-size:16px'>반려견 배상책임 "
 "특별약관</h1><h1 id='15' style='font-size:14px'>제1조(목적)</h1><br><p id='16'"),
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
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
