from langchain_core.documents import Document

chunk = Document(
    page_content=('※유의사항 관련 예시\n'
 'A씨(피보험자)는 일반 사무직으로 근무하던 중 상해보험을 가입하고 몇 년 후 물품배달원으로 직업을 변경하였으나 이를 고의 또는 중대한 '
 '과실로 보험회사에 알리지 않았고, 물품 배달 업무 중 일반상해로 사고가 발생한 후 보험금을 청구하였으나 보험금이 약정한 보험금보다 적게 '
 '지급되었 습니다.\n'
 '제15조 (알릴 의무 위반의 효과)\n'
 '① 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 이 계약을 해\n'
 '반하고 그 의무가 중요한 사항에 해당하는 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
