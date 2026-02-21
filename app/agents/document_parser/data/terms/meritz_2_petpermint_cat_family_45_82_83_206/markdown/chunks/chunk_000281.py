from langchain_core.documents import Document

chunk = Document(
    page_content=('- 실험 및 이와 유사한 목적으로 이용함으로써 발생한\n'
 '- 손해\n'
 '- ⑨ 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의\n'
 '- 사 자격이 없는 자의 치료행위로 인한 비용 및 그로\n'
 '- 인하여 가중된 비용\n'
 '- ⑩ 국가 및 지방자치단체의 명령 또는 법률에 의한 살처\n'
 '- 분 또는 이와 유사한 사태\n'
 '- ⑪ 대한민국 이외의 지역에서 발생한 사고 및 손해\n'
 '\uf000 회사는 다음에 정한 사유 중 하나에 의해 피보험자가 부\n'
 '담한 치료비, 비용 또는 손해에 대해서는 보험금을 지급하\n'
 '지 않습니다.- ① 반려동물의 선천적, 유전적 질병에 의한 손해(보험계'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
