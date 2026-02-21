from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험개발원이 공시하는 보험계약대출이율: 보험개발원이 정기적으로 산출하여 공시<br>하는 이율로써 회사가 보험금의 지급 또는 보험료의 '
 "환급을 지연하는 경우 등에 적<br>용합니다.</p><br><h1 id='17' style='font-size:14px'>5. 기간과 "
 "날짜 관련 용어</h1><br><p id='18' data-category='list' style='font-size:14px'>가. "
 '보험기간: 계약에 따라 보장을 받는 기간을 말합니다.<br>나'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000012',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
