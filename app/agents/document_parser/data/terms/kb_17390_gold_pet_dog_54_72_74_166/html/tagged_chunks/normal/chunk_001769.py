from langchain_core.documents import Document

chunk = Document(
    page_content=('되는 항목</td><td>분류번호</td></tr><tr><td>폐렴연쇄알균에 의한 폐렴 '
 '폐렴</td><td>J13</td></tr><tr><td rowspan="2">특정외부요인 폐질환</td><td>인플루엔자균에 의한 폐렴 '
 '화학물질, 가스, 훈증기 및 물김의</td><td>J14</td></tr><tr><td>흡입에 의한 '
 '호흡기병태</td><td>J68</td></tr><tr><td rowspan="2">하부호흡기</td><td>고체 및 액체에 의한'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001769',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
