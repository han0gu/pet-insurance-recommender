from langchain_core.documents import Document

chunk = Document(
    page_content=('- 분 또는 이와 유사한 사태\n'
 '- ⑪ 대한민국 이외의 지역에서 발생한 사고 및 손해\n'
 '\uf000 회사는 다음에 정한 사유 중 하나에 의해 피보험자가 부\n'
 '담한 치료비, 비용 또는 손해에 대해서는 보험금을 지급하\n'
 '지 않습니다.- ① 반려동물의 선천적, 유전적 질병에 의한 손해(보험계\n'
 '- 약 이전부터 객관적으로 인지할 수 있는 증상을 포함\n'
 '- 합니다. 다만, 보험기간 중 최초로 발견된 경우에는\n'
 '- 해당 보험기간에 한하여 보상합니다.)\n'
 '- ② 다음 정한 질병 및 이에 기인하는 질병(다만, 질병의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000446',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
