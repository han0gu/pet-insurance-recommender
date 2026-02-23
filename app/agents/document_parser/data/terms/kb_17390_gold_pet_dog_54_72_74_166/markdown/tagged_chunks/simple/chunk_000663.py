from langchain_core.documents import Document

chunk = Document(
    page_content=('| 보험금 분담 | 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)가 있을 경우 비율에 따라 손 해를 '
 '보상합니다. |\n'
 '| 대위권 | 회사가 보험금을 지급하고 취득하는 법률상의 권리를 말합 니다. |\n'
 '| 배상책임 | 보험증권상의 보장지역 내에서 보험기간 중에 발생된 보험 사고로 인하여 타인에게 입힌 손해에 대한 법률상의 책임을 '
 '말합니다. |\n'
 '- 120 -용 어 정 의 부 가 설 명\n'
 '∙ 핵연료물질 : 사용된 연료를 포함합니다.\n'
 '배상책임에 있어 회사와 계약자간에 약정한 금액으로 피보 특'),
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
 'indexing': {'chunk_id': 'chunk_000663',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
