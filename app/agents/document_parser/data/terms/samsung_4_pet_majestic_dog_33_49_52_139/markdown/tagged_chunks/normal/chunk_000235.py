from langchain_core.documents import Document

chunk = Document(
    page_content=('- 60 -계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일\n'
 '단 , 계약해당일 2월 29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.# 제 25조 (특별약관의 소멸)각 특별약관의 보장을 '
 '따릅니다.제5관 보험료의 납입# 제 26조 (제1회 보험료 및 회사의 보장개시)- ① 회사는 계약의 청약을 승낙하고 제1회 보험료를 받은 '
 '때부터 이 약관이 정한 바에 따\n'
 '- 라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료를 받은 후 승낙한 경우에'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000235',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
