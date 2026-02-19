from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[비례 보상]\n'
 '보험기간 중 직업의 변경으로 위험이 증가(상해급수 1급 → 2급)되었으나, 이를 회사에 알리지 않고 변경전 보험료를 계속 납입하던 중 '
 '상해사망 사고가 발생한 경우\n'
 '∙ 상해사망 가입금액 : 1억원 ∙ 상해사망 보험요율 : 1급 0.3, 2급 0.5 → 고객이 수령하는 상해사망 보험금 = 1억원 × '
 '(0.3 ÷ 0.5) = 6천만원'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 49},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000194',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
