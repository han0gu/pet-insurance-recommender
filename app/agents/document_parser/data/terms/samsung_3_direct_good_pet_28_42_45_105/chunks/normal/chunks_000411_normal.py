from langchain_core.documents import Document

chunk = Document(
    page_content=('① 이 특별약관에서의 반려견의 나이는 만나이를 기준으로 합니다. ② 제1항의 만나이는 계약일 현재 반려견의 실제 만나이를 기준으로 하며, '
 '이후 매년 계 약해당일에 나이가 증가하는 것으로 합니다. ③ 반려견의 나이 및 품종에 관한 청약서상 기재사항이 사실과 다른 경우에는 '
 '정정된 나 이 및 품종에 해당하는 보험금 및 보험료로 변경합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000411',
              'chunk_char_len': 186,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
