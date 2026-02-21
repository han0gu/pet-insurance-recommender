from langchain_core.documents import Document

chunk = Document(
    page_content=('지급사유가 발생하는 때에 회사에 보험금을 청구하 여 받을 수 있는 사람을 '
 '말합니다.</td></tr><tr><td>보험증권</td><td>계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자 에게 드리는 '
 '증서를 말합니다.</td></tr><tr><td>피보험자</td><td>보험사고로 인하여 타인에 대한 법률상 손해배상책임을 부 담하는 '
 '손해를 입은 사람(법인인 경우에는 그 이사 또는 법 인의 업무를 집행하는 그 밖의 기관)을 '
 "말합니다.</td></tr></tbody></table><p id='161'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001141',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
