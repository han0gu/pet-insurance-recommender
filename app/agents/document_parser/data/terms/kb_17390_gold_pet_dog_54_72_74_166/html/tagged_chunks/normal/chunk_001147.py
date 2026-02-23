from langchain_core.documents import Document

chunk = Document(
    page_content=('계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)가 있을 경우 비율에 따라 손 해를 '
 '보상합니다.</td></tr><tr><td>대위권</td><td>회사가 보험금을 지급하고 취득하는 법률상의 권리를 말합 '
 '니다.</td></tr><tr><td>배상책임</td><td>보험증권상의 보장지역 내에서 보험기간 중에 발생된 보험 사고로 인하여 '
 "타인에게 입힌 손해에 대한 법률상의 책임을 말합니다.</td></tr></tbody></table><p id='165' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001147',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
