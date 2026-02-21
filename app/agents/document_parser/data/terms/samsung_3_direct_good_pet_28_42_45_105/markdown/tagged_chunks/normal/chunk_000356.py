from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내># [계약해당일 계산]최초계약일과 동일한 월, 일을 말합니다.\n'
 '계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일\n'
 '단, 계약해당일 2월 29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.# 제19조 (특별약관의 소멸)① 보험증권에 기재된 '
 '반려견이 보험기간 중에 사망하여 보험의 목적에 대해 이 특별약\n'
 '관에서 정한 보험금 지급사유가 더이상 발생할 수 없는 경우에는 “보험료 및 해약환\n'
 '급금 산출방법서”에 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000356',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
