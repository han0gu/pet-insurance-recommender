from langchain_core.documents import Document

chunk = Document(
    page_content=("연료를 포함합니다.</h1><br><h1 id='45' style='font-size:20px'>【핵연료물질에 의하여 오염된 "
 "물질】</h1><br><h1 id='46' style='font-size:20px'>원자핵분열 생성물을 포함합니다.</h1><br><p "
 "id='47' data-category='list' style='font-size:16px'>⑥ 최초 계약의 보험계약일 이전에 이미 감염 "
 '또는 발병<br>한 질병 및 상해<br>⑦ 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또<br>는 급수 등 기본적인 관리에 대한'),
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
 'indexing': {'chunk_id': 'chunk_000455',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
