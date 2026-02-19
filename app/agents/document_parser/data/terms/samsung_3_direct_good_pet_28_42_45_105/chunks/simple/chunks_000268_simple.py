from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[강제집행과 담보권실행]\n'
 '강제집행이란 사법상 또는 행정법상의 의무를 이행하지 않는 사람에 대하여 국가가 강제 권력으로 그 의무를 이행하는 것을 말합니다. '
 '담보권실행이란 담보권을 설정한 채권자가 채무를 이행하지 않는 채무자에 대하여 해당 담보권을 실행하는 것을 말합니다. 법원은 채권자의 '
 '신청에 따른 강제집행 및 담보권실행으로 채무자의 해약환급금을 압류할 수 있으며, 법원의 추심명령 또는 전부명령에 따라 회사는 채권자에게 '
 '해약환급금을 지급하게 됩니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 56},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000268',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
