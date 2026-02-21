from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기 전에 적용된 보험요율(이하「변경전 요율」이라 합니다)의 위험이 증가된 후에 적\n'
 '- 용해야 할 보험요율(이하「변경후 요율」이라 합니다)에 대한 비율에 따라 보험금을\n'
 '- 삭감하여 지급합니다. 다만, 증가된 위험과 관계없이 발생한 보험금 지급사유에 관해\n'
 '- 서는 원래대로 지급합니다.\n'
 '<예시안내># [비례 보상]보험기간 중 직업의 변경으로 위험이 증가(상해급수 1급 → 2급)되었으나, 이를 회사에 알리지\n'
 '않고 변경전 보험료를 계속 납입하던 중 상해사망 사고가 발생한 경우- ∙ 상해사망 가입금액 : 1억원'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000166',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
