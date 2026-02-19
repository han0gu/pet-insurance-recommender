from langchain_core.documents import Document

chunk = Document(
    page_content=('서는 원래대로 지급합니다.\n'
 '<예시안내>\n'
 '[비례 보상]\n'
 '보험기간 중 직업의 변경으로 위험이 증가(상해급수 1급 → 2급)되었으나, 이를 회사에 알리지 않 고 변경전 보험료를 계속 납입하던 중 '
 '상해사망 사고가 발생한 경우\n'
 '∙ 상해사망 가입금액 : 1억원 ∙ 상해사망 보험요율 : 1급 0.3, 2급 0.5\n'
 '→ 고객이 수령하는 상해사망 보험금 = 1억원 × (0.3 ÷ 0.5) = 6천만원'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 39},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000069',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
