from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조 (보험금의 청구)\n'
 '지정대리청구인은 회사가 정하는 방법에 따라 다음의 서류를 제출하고 보험금을 청구하 여야 합니다.\n'
 '1. 청구서(회사양식) 2. 사고증명서(장해진단서, 입원치료확인서 등) 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 '
 '발행 신분증) 4. 피보험자 및 지정대리청구인의 가족관계등록부(가족관계증명서) 5. 기타 지정대리청구인이 보험금의 수령에 필요하여 '
 '제출하는 서류\n'
 '제7조 (준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.\n'
 '- 100 -\n'
 '100 / 181'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000640',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
