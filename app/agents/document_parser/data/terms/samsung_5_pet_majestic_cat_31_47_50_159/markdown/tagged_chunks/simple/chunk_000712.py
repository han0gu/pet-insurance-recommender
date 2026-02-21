from langchain_core.documents import Document

chunk = Document(
    page_content=('| 58 | 연골증 | M94 | 연골의 기타 장애 |\n'
 '주 ) 제10차 개정 이후 한국표준질병·사인분류에서 상기 분류표에 변경사항이 발생하는 경우에는\n'
 '변경된 분류표에 따릅니다.- 130 -# 5-5. 보험료 자동납입 특별약관# 제1조 (보험료납입)- ① 보험계약자(이하 「계약자」 라 '
 '합니다)는 제2회 이후의 보험료부터 이 특별약관에 따\n'
 '- 라 계약자의 지정계좌를 이용하여 보험료를 자동납입하거나 급여이체를 통하여 납입\n'
 '- 합니다.\n'
 '- ② 제1회 보험료의 납입방법을 계약자의 지정 금융기관 지정계좌를 통한 자동납입으로'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000712',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
