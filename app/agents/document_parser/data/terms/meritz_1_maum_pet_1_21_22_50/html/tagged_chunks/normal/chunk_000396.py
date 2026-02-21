from langchain_core.documents import Document

chunk = Document(
    page_content=(". 보험계약대출이율은 보험개발원이 공시하는 보험계약대출이율을 적용합니다.</p><footer id='85' "
 "style='font-size:14px'>- 48 -</footer><caption id='86' "
 "style='font-size:14px'><부표2> 보험금을 지급할 때의 적립이율</caption><p id='87' "
 "data-category='paragraph' style='font-size:14px'>(배상책임 특별약관 제7조 제2항 "
 "관련)</p><table id='88'"),
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
 'indexing': {'chunk_id': 'chunk_000396',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
