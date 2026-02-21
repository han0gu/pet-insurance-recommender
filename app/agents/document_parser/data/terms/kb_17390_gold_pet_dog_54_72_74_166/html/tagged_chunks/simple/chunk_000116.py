from langchain_core.documents import Document

chunk = Document(
    page_content=('. ※ 유의사항 관련 예시: A씨(피보험자)는 일반 사무직으로 근무하던 중 상해보 험을 가입하고 몇 년 후 물품배달원으로 직업을 '
 '변경하였으나 이를 고의 또는 중대한 과실로 보험회사에 알리지 않았고, 물품 배달 업무 중 일반상해로 사 고가 발생한 후 보험금을 '
 '청구하였으나 보험금이 약정한 보험금보다 적게 지</td></tr></tbody></table></td></tr><tr><td '
 'colspan="2">급되었습니다.</td></tr><tr><td>용 어 풀 이</td><td>중대한 과실</td></tr><tr><td'),
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
 'indexing': {'chunk_id': 'chunk_000116',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
