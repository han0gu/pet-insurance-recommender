from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1988년 10월 2일\n'
 '33년 6개월 11일 = 34세[계약해당일 계산]\n'
 '최초계약일과 동일한 월, 일을 말합니다.- 58 -계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일\n'
 '단 , 계약해당일 2월 29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.# 제 25조 (특별약관의 소멸)각 특별약관의 보장을 '
 '따릅니다.제5관 보험료의 납입# 제 26조 (제1회 보험료 및 회사의 보장개시)- ① 회사는 계약의 청약을 승낙하고 제1회 보험료를 받은 '
 '때부터 이 약관이 정한 바에 따'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000234',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
