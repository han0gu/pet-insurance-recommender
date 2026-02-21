from langchain_core.documents import Document

chunk = Document(
    page_content=('- 타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여\n'
 '- 보상한다.\n'
 '- 파) 외상후 스트레스장애, 우울증(반응성) 등의 질환,\n'
 '- 정신분열증(조현병), 편집증, 조울증(양극성장\n'
 '- 애), 불안장애, 전환장애, 공포장애, 강박장애 등\n'
 '- 각종 신경증 및 각종 인격장애는 보상의 대상이\n'
 '- 되지 않는다.\n'
 '# 3) 치매가) “치매”라 함은 정상적으로 성숙한 뇌가 질병이\n'
 '나 외상 후 기질성 손상으로 파괴되어 한번 획득\n'
 '한 지적기능이 지속적 또는 전반적으로 저하되는\n'
 '것을 말한다.나) 치매의 장해평가는 임상적인 증상 뿐 아니라 뇌영'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000617',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
