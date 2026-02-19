from langchain_core.documents import Document

chunk = Document(
    page_content=('제도성 특별약관\n'
 '※ 약관에서 인용된 법·규정은 「별표 및 참고」 의 「약관에서 인용된 법·규정」 에서 확인할 수 있습니다.\n'
 '제도성 특별약관\n'
 '제1조 (적용대상)\n'
 '이 특별약관은 손해의 보상을 내용으로 하는 이 계약의 다른 특별약관 중 [갱신형] 특별 약관(이하 「갱신형 계약」 이라 합니다.)에 '
 '대하여 적용합니다.\n'
 '제2조 (자동갱신에 관한 사항)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000613',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
