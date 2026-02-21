from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 성별에 해당하는 보험금 및 보험료로 변경합니다.\n'
 '<예시안내>[보험나이 계산]\n'
 '생년월일 : 1988년 10월 2일예1) 계 약 일 : 2022년 3월 13일⇒ 2022년 3월 13일\n'
 '- 1988년 10월 2일\n'
 '33년 5개월 11일 = 33세예 2) 계 약 일 : 2022년 4월 13일- ⇒ 2022년 4월 13일\n'
 '- - 1988년 10월 2일\n'
 '- 33년 6개월 11일 = 34세'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
