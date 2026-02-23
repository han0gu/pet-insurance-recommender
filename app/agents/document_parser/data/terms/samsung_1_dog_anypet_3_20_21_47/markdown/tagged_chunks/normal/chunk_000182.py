from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보험료 납입영수증에 장애인전용 보장성보험료로 표시됩니다.\n'
 '# 【예시】2019년 1월 15일에 전환대상계약에 가입한 계약자가 2019년 6월 1일에 이 특별약관을 청약하고 회사\n'
 '가 승낙하여 전환대상계약이 장애인전용보험으로 전환된 경우, 이 특별약관을 청약하기 전(2019년 1월\n'
 '15일~ 2019년 5월 31일)에 납입된 보험료는 해당 연도 보험료 납입영수증에 장애인전용 보장성 보험\n'
 '료로 표시되지 않고 특별세액공제 대상에 포함되지 않으며, 장애인전용보험으로 전환된 이후(2019년6'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000182',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
