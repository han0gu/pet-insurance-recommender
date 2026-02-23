from langchain_core.documents import Document

chunk = Document(
    page_content=('회사와 계약자가 합의하여 관할법원을 달리 정할 수 있습니다.# 제38조 (소멸시효)보험금청구권, 만기환급금청구권, 보험료 반환청구권, '
 '해약환급금 청구권, 계약자적립액\n'
 '및 미경과보험료 반환청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다.<용어풀이>[소멸시효]소멸시효는 해당 청구권을 행사할 수 있는 '
 '때부터 진행합니다. 보험금 지급사유가 2021년 4월 1일에# 제39조 (약관의 해석)① 회사는 신의성실의 원칙에 따라 공정하게 약관을 '
 '해석하여야 하며 계약자에 따라 다'),
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
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000117',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
