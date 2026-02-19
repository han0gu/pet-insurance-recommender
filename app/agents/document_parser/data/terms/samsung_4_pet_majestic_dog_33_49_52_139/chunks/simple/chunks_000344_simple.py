from langchain_core.documents import Document

chunk = Document(
    page_content=('제2관 개별사항\n'
 '제1조 (보험금의 지급사유)\n'
 '회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합 니다) 중에 상해로 장해분류표([별표2]장해분류표 '
 '참조. 이하 같습니다)에서 정한 3~100% 장해지급률에 해당하는 장해상태가 되었을 때에는 장해분류표에서 정한 지급률 을 보험증권에 '
 '기재된 이 특별약관의 보험가입금액(이하「상해 후유장해보험가입금액」 이라 합니다)에 곱하여 산출한 금액을 상해 후유장해보험금으로 '
 '보험수익자에게 지급합 니다.\n'
 '<용어풀이>\n'
 '[장해지급률]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 69},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000344',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
