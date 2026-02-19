from langchain_core.documents import Document

chunk = Document(
    page_content=('⑥ 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때 다음 각 호의 요건을 모두 충족하는 경우에 '
 '「보험업감독규정」 제4-36조 제3항에 따른 전자적 상품설명장치를 활용할 수 있습니다.\n'
 '1.계약자에게 전자적 상품설명장치를 활용하여 제1항에 따른 납입최고(독촉) 등을 한다 는 사실을 미리 안내하고 동의를 받을 것'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000423',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
