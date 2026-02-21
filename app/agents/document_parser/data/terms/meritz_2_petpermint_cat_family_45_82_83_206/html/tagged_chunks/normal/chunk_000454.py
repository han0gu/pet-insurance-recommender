from langchain_core.documents import Document

chunk = Document(
    page_content=('혁명, 내란, 사변, 폭동, 소<br>요, 기타 이들과 유사한 사태<br>④ 핵연료물질 또는 핵연료물질에 의하여 오염된 '
 '물질의<br>방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의<br>특성에 의한 사고<br>⑤ 제4호 이외의 방사선을 쬐는 것 또는 '
 "방사능 오염</p><br><h1 id='43' style='font-size:20px'>【핵연료물질】</h1><br><h1 id='44' "
 "style='font-size:20px'>사용된 연료를 포함합니다.</h1><br><h1 id='45'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000454',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
