from langchain_core.documents import Document

chunk = Document(
    page_content=('폐렴</td><td>J69</td></tr><tr><td rowspan="3">중금속에 의한 질환</td><td>기타 외부요인에 의한 '
 '호흡기 병태 약물 및 중금속 유발 세뇨관-간질 및 세뇨관 병태</td><td>J70 N14 '
 '특별</td></tr><tr><td>정상적으로는 혈액내에 없는 약물 및</td><td>약 R78 '
 '관</td></tr><tr><td>기타물질의 소견 금속의 독성효과</td><td>T56</td></tr><tr><td '
 'colspan="3"></td></tr></tbody></table><br><p id=\'110\''),
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
 'indexing': {'chunk_id': 'chunk_001795',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
