from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하 며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 '
 '부담합니다. ⑤ 같은 상해로 두 가지 이상의 후유장해가 생긴 경우에는 후유장해 지급률을 합산하여 지급합니다. 다만, 장해분류표의 각 '
 '신체부위별 판정기준에 별도로 정한 경우에는 그 기준에 따릅니다. ⑥ 다른 상해로 인하여 후유장해가 2회 이상 발생하였을 경우에는 그 '
 '때마다 이에 해당 하는 후유장해지급률을 결정합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 29},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
