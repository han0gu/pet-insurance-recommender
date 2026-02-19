from langchain_core.documents import Document

chunk = Document(
    page_content=('. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 '
 '또는 안전성과 신뢰성이 확보된 전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함) 6. 수탁기관 위탁비용 영수증 및 '
 '동물관리위탁업자가 제공하는 계약서(위탁관리업소 등록번호, 업소명 및 주소, 전화번호, 위탁관리동물 종류, 품종, 나이, 서비스 기간, '
 '비용 등 포함) 7. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 91},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000593',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
