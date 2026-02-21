from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그리고 만기환급금 지급 시기에 만기환급금을 청구하여 받을 수 있는 사람을 말합 '
 '니다.</td></tr><tr><td>보험증권</td><td>계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자 에게 드리는 증서를 '
 '말합니다.</td></tr><tr><td>진단계약</td><td>계약을 체결하기 위하여 피보험자가 건강진단을 받아야 하 는 계약을 '
 '말합니다.</td></tr><tr><td>피보험자</td><td>보험사고의 대상이 되는 사람을 '
 "말합니다.</td></tr></tbody></table><p id='1'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000003',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
