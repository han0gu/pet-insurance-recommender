from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[보험안내자료]\n'
 '계약의 청약을 권유하기 위해 만든 자료 등을 말합니다. [기명날인] 자기 이름을 쓰고 도장을 찍는 것을 말합니다.\n'
 '제41조 (회사의 손해배상책임)\n'
 '① 회사는 계약과 관련하여 임직원, 보험설계사 및 대리점의 책임있는 사유로 계약자, 피 보험자 및 보험수익자에게 발생된 손해에 대하여 '
 '관계 법령 등에 따라 손해배상의 책 ( - 、'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000142',
              'chunk_char_len': 198,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
