from langchain_core.documents import Document

chunk = Document(
    page_content=('- 된 경우 확대된 내용에 대해 회사는 재가입 시점의 인수기준에 따라 승낙하거나 일부\n'
 '- 보장을 제한할 수 있습니다.\n'
 '- ③ 회사는 계약자에게 재가입주기(보장내용 변경주기)가 끝나는 날 이전까지 2회 이상 재\n'
 '가입 요건, 보장내용 변경내역, 보험료 수준, 재가입 절차 및 재가입 의사 여부를 확인하는 내용 등을 서면(등기우편 등), '
 '전화(음성녹음), 전자문서, 휴대전화 문자메시지또는 이에 준하는 전자적 의사표시 등으로 알려드리고, 회사는 계약자의 재가입의사'),
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
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000384',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
