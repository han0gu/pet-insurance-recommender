from langchain_core.documents import Document

chunk = Document(
    page_content=('제 17조 (특별약관 내용의 변경 등)\n'
 '① 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 있습니다. 이 경우 승낙을 서 면 등으로 알리거나 보험증권의 뒷면에 기재하여 '
 '드립니다.\n'
 '1. 보험종목 2. 보험기간 3. 보험료 납입주기, 납입방법 및 납입기간 4. 계약자, 피보험자, 반려견 5. 보험가입금액(배상책임의 '
 '경우 보상한도액) 등 기타 계약의 내용'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000600',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
